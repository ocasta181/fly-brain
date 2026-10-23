"""Preparation-only lowering of Brian's optimized C++ arithmetic.

Brian supplies typed statements and its C++ generator; the installed Clang
optimizer supplies the operation DAG. Execution remains integer/software
binary64 Metal. No reference executable or CPU simulation is run here.
The NumPy arithmetic policy is independent of this explicit CPU policy.

The operation graph is reused from the qualified shared-state checkpoint
0b53bb3. Optional operator_registry hooks preserve that graph's arithmetic and
add target-specific source intrinsics, immutable folding and error transport.
"""
from __future__ import annotations

from collections import Counter
import ast
from copy import copy
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sysconfig

import numpy as np

from .compiler import CompiledVariable, ScalarStage, UnsupportedModelError, _literal


_TYPE = {"double": ("float64", "ulong"), "i64": ("int64", "long"),
         "i32": ("int32", "int"), "i16": ("int16", "short"),
         "i8": ("int8", "char"), "i1": ("bool", "bool")}
_REF = r"%[-\w.$]+"
_TERMINATING_ERROR = re.compile(
    r"(?:tail )?call void @_eon_arithmetic_(scalar|vector)_error\(i32 (?:noundef )?([^),]+)\)")


def _pointwise(line):
    # Brian's vector block reads/writes its own row and cannot write shared
    # state. LLVM vector arithmetic is therefore the same scalar operation
    # independently in every lane. Only uniform constants and scalar splats
    # are accepted here; permutations are handled/validated separately.
    line = re.sub(r"<\d+ x (double|i\d+)>", r"\1", line)
    line = re.sub(r"splat \((?:double|i\d+) ([^)]+)\)", r"\1", line)
    return re.sub(r"\.v\d+(f64|i\d+)", r".\1", line)


@lru_cache(maxsize=8)
def _version(command):
    result = subprocess.run([*command, "--version"], capture_output=True, text=True, check=True, timeout=10)
    if "clang" not in result.stdout.lower():
        raise UnsupportedModelError("Optimized C++ arithmetic requires its Clang LLVM operation graph")
    return result.stdout


def compiler_identity(target="cpp_standalone", *, flags_override=None, python_flags=None):
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    _, flags = get_compiler_and_args()
    if flags_override is not None:
        flags = list(flags_override)
    if target == "cython":
        # Cython extensions inherit Python's compiler flags before Brian's
        # flags (notably signed integer wrapping). Preserve that ordering.
        inherited = (list(python_flags) if python_flags is not None else
                     shlex.split(sysconfig.get_config_var("CFLAGS") or "")+
                     shlex.split(os.environ.get("CFLAGS", "")))
        inherited += shlex.split(os.environ.get("CPPFLAGS", ""))
        flags = inherited + list(flags)
    command = shlex.split(os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "clang++")
    executable = shutil.which(command[0])
    if executable is None:
        raise UnsupportedModelError("C++ arithmetic compiler is unavailable")
    command[0] = os.path.realpath(executable)
    identity = (tuple(command), tuple(flags), _version(tuple(command)))
    return identity


@lru_cache(maxsize=64)
def _optimize(source, identity):
    command, flags, version = identity
    result = subprocess.run([*command, *flags, "-x", "c++", "-", "-S", "-emit-llvm", "-o", "-"],
                            input=source, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise UnsupportedModelError("C++ arithmetic graph compilation failed: " + result.stderr)
    return result.stdout


def _source(resolved, scalar, vector, group, runner, phase, neurons, target,operator_registry=None):
    """Use Brian's actual read/write rules, function support and C++ renderer."""
    from collections import defaultdict
    from brian2.codegen.generators.cpp_generator import CPPCodeGenerator
    from brian2.core.variables import ArrayVariable, Constant
    from brian2.devices.device import get_device
    from brian2.utils.stringtools import word_substitute
    from brian2.parsing.rendering import CPPNodeRenderer
    from .symbols import variable_code_name, variable_code_names
    data_names = {name for name,var in resolved.items() if hasattr(var,"dtype")}
    data_names.update(statement.var for statement in scalar+vector)
    symbols = variable_code_names(data_names)

    class RegistryRenderer(CPPNodeRenderer):
        def render_Name(self,node):
            return symbols.get(node.id,super().render_Name(node))

        def render_func(self,node):
            return CPPNodeRenderer.render_Name(self,node)

        def render_Call(self,node):
            if operator_registry is not None:
                code=operator_registry.source_call(self,node)
                if code is not None:return code
            return super().render_Call(node)

    class CythonDivisionRenderer(RegistryRenderer):
        def render_BinOp(self, node):
            if operator_registry is not None:
                code=operator_registry.source_binary(self,node)
                if code is not None:return code
            if isinstance(node.op, ast.Div):
                role = "scalar" if self.scalar_statement else "vector"
                self.division_roles.add(role)
                kind=f', _eon_{role}_error_kind' if operator_registry is not None else ''
                return f"_eon_cython_div({self.render_node(node.left)}, {self.render_node(node.right)}, _eon_{role}_error{kind})"
            return super().render_BinOp(node)

    class SymbolGenerator(CPPCodeGenerator):
        def translate_statement(self,statement):
            statement=copy(statement)
            statement.var=symbols.get(statement.var,statement.var)
            if statement.used_boolean_variables:
                statement.used_boolean_variables=[symbols.get(name,name)
                    for name in statement.used_boolean_variables]
                statement.boolean_simplified_expressions={
                    tuple((symbols.get(name,name),value) for name,value in conditions):expression
                    for conditions,expression in statement.boolean_simplified_expressions.items()}
            return super().translate_statement(statement)

        def translate_to_read_arrays(self,read,write,indices):
            # The declaration's final token is its local data name. Type and
            # pointer tokens remain untouched, even for a field named uint.
            lines=super().translate_to_read_arrays(read,write,indices)
            return [re.sub(r"\b([A-Za-z_][A-Za-z_0-9]*)(?=\s*=)",
                lambda match:symbols.get(match[1],match[1]),line) for line in lines]

        def translate_to_declarations(self,read,write,indices):
            lines=super().translate_to_declarations(read,write,indices)
            return [re.sub(r"\b([A-Za-z_][A-Za-z_0-9]*)(?=\s*;)",
                lambda match:symbols.get(match[1],match[1]),line) for line in lines]

        def translate_to_write_arrays(self,write):
            lines=super().translate_to_write_arrays(write)
            return [re.sub(r"(=\s*)([A-Za-z_][A-Za-z_0-9]*)(\s*;)$",
                lambda match:match[1]+symbols.get(match[2],match[2])+match[3],line) for line in lines]

        def translate_to_statements(self,statements,conditional_write_vars):
            lines=super().translate_to_statements(statements,conditional_write_vars)
            return [re.sub(r"^if\(([A-Za-z_][A-Za-z_0-9]*)\)$",
                lambda match:"if("+symbols.get(match[1],match[1])+")",line) for line in lines]

    class CythonDivisionGenerator(SymbolGenerator):
        @staticmethod
        def c_data_type(dtype):
            # The Cython table includes char-backed uint8 arrays and true
            # C++ bools. CPP's table is a different accepted source contract.
            from brian2.codegen.generators.cython_generator import get_cpp_dtype
            return get_cpp_dtype(np.dtype(dtype))

        def translate_expression(self, expr):
            if operator_registry is not None:operator_registry.bind_function_aliases(self.func_name_replacements)
            expr = word_substitute(expr, self.func_name_replacements)
            renderer_class=CythonDivisionRenderer if target=='cython' else RegistryRenderer
            renderer = renderer_class(auto_vectorise=self.auto_vectorise)
            renderer.scalar_statement = getattr(self, "scalar_statement", False)
            renderer.division_roles = self.division_roles
            renderer.augmented_mod=getattr(self,'augmented_mod',False)
            return renderer.render_expr(expr).strip()

        def translate_statement(self, statement):
            self.scalar_statement = statement.scalar
            self.augmented_mod=target=='cython' and statement.op=='%='
            if target=='cython' and statement.op in ("/=","%="):
                statement = copy(statement)
                operator='/' if statement.op=='/=' else '%'
                statement.op = "="
                statement.expr = f"{statement.var}{operator}({statement.expr})"
                statement.used_boolean_variables = None
            try:
                return super().translate_statement(statement)
            finally:
                self.augmented_mod=False

    indices = defaultdict(lambda: "_idx", {name: "0" for name, var in resolved.items()
                                         if getattr(var, "scalar", False)})
    resolved = dict(resolved)
    if target == "cython":
        resolved.setdefault("N", group.variables["N"])
    generator_class = CythonDivisionGenerator if target == "cython" or operator_registry is not None else SymbolGenerator
    generator = generator_class(dict(sorted(resolved.items())), indices, group, {"_idx"}, CPPCodeGenerator,
        "eon_cpu_" + phase, runner.template, allows_scalar_write=phase in ("state", "subexpression"),
        override_conditional_write={"not_refractory"} if phase == "reset" else None)
    if target=='cython':
        # CPPCodeGenerator.__init__ installs its module-level mapper on the
        # instance, so restore the Cython mapper after that initialization.
        generator.c_data_type=CythonDivisionGenerator.c_data_type
    generator.division_roles = set()
    sc, ve, keywords = generator.translate_statement_sequence({"": scalar}, {"": vector})
    arrays = {get_device().get_array_name(var): (name, var)
              for name, var in resolved.items() if isinstance(var, ArrayVariable)}
    declarations, pointers, rename = [], [], {}
    inputs = {}
    for role in sorted(generator.division_roles):
        global_name = f"_eon_error_array_{role}"
        declarations.append(f"extern bool* {global_name};")
        pointers.append(f"bool* __restrict _eon_{role}_errors = {global_name};")
        inputs[global_name] = f"_eon_{role}_error"
        if operator_registry is not None:
            name=f'_eon_kind_array_{role}'
            declarations.append(f'extern int32_t* {name};')
            pointers.append(f'int32_t* _eon_{role}_error_kinds={name};')
            inputs[name]=f'_eon_{role}_error_kind'
    for original, (name, var) in sorted(arrays.items()):
        global_name = "_eon_array_" + name
        pointer = "_ptr" + global_name
        rename[original], rename["_ptr" + original] = global_name, pointer
        ctype = generator.c_data_type(var.dtype)
        declarations.append(f"extern {ctype}* {global_name};")
        restrict = "__restrict" if target != "cython" and not var.scalar and neurons > 1 else ""
        pointers.append(f"{ctype}* {restrict} {pointer} = {global_name};")
        inputs[global_name] = name
    constants, runtime_constants = [], {}
    for name, var in sorted(resolved.items()):
        if isinstance(var, Constant):
            value = neurons if name == "N" else np.asarray(var.get_value()).item()
            ctype = generator.c_data_type(var.dtype)
            if target == "cython":
                from brian2.codegen.generators.cython_generator import get_cpp_dtype
                # Brian's Cython generator declares Constants from value,
                # which may be Python float despite float32 var.dtype.
                ctype=get_cpp_dtype(var.value)
                global_name = "_eon_constant_" + name
                declarations.append(f"extern {ctype} {global_name};")
                inputs[global_name] = name
                runtime_constants[name] = value
                constants.append(f"const {ctype} {symbols.get(name,name)} = {global_name};")
            else:
                literal = str(value).lower() if isinstance(value, bool) else repr(value)
                constants.append(f"const {ctype} {symbols.get(name,name)} = {literal};")
    if "N" not in resolved:
        constants.append(f"const int N = {neurons};")
    vector_code = "\n".join(ve[""])
    if phase == "threshold":
        declarations.append("extern char* _eon_conditions;")
        inputs["_eon_conditions"] = "_cond"
        vector_code += "\n_eon_conditions[_idx] = _cond;"
    if phase == "reset":
        # Brian visits the spike-index list, not an unconditional dense range.
        declarations.append("extern int* _eon_spikes;")
        inputs["_eon_spikes"] = "i"
        loop = "for(int _j=0; _j<_eon_spikes[N]; ++_j) { int _idx=_eon_spikes[_j];"
    else:
        loop = "for(int _idx=0; _idx<N; ++_idx) {"
    source = "\n".join([
        "#include <cstdint>", "#include <cmath>", "#include <algorithm>", "using namespace std;",
        *( (operator_registry.support_code,) if operator_registry is not None else () ),
        *((operator_registry.division_support_code,) if operator_registry is not None and target=='cython' else
          ('inline double _eon_cython_div(double a, double b, bool& error) { error = error | (b == 0.); return a/b; }',)
          if target == "cython" else ()),
        "template<typename T> inline int int_(T value) { return (int)value; }",
        *keywords["hashdefine_lines"], *keywords["support_code_lines"], 'extern "C" {',
        *declarations, f"void eon_cpu_{phase}() {{", *constants, *pointers,
        *(('bool _eon_scalar_error=false;',) if 'scalar' in generator.division_roles else ()),
        *(('int32_t _eon_scalar_error_kind=0;',) if operator_registry is not None and 'scalar' in generator.division_roles else ()),
        *sc[""],
        *(('_eon_scalar_errors[0]=_eon_scalar_error;',) if 'scalar' in generator.division_roles else ()),
        *(('_eon_scalar_error_kinds[0]=_eon_scalar_error_kind;',) if operator_registry is not None and 'scalar' in generator.division_roles else ()),
        loop, "const int _vectorisation_idx=_idx;",
        *(('bool _eon_vector_error=false;',) if 'vector' in generator.division_roles else ()),
        *(('int32_t _eon_vector_error_kind=0;',) if operator_registry is not None and 'vector' in generator.division_roles else ()),vector_code,
        *(('_eon_vector_errors[_idx]=_eon_vector_error;',) if 'vector' in generator.division_roles else ()), "}", "}", "}",
    ])
    source = word_substitute(source, rename)
    writes = {s.var for s in scalar + vector if s.var in resolved and isinstance(resolved[s.var], ArrayVariable)}
    if phase == "threshold":
        writes.add("_cond")
    writes.update(f"_eon_{role}_error" for role in generator.division_roles)
    if operator_registry is not None:
        source=source.replace('_eon_vector_errors[_idx]=_eon_vector_error;',
            '_eon_vector_errors[_idx]=_eon_vector_error;_eon_vector_error_kinds[_idx]=_eon_vector_error_kind;')
        writes.update(f'_eon_{role}_error_kind' for role in generator.division_roles)
    return source, inputs, writes, runtime_constants


@dataclass(eq=False)
class _Node:
    op: str
    args: tuple
    type: str
    shared: bool
    value: object = None
    flags: str = ""


def _kind(type):
    return "float" if type == "double" else "bool" if type == "i1" else "int"


class _Graph:
    def __init__(self, ir, inputs, writes, inventory, model, phase, neurons, runtime_constants=None,operator_registry=None):
        self.operator_registry=operator_registry
        self.ir, self.inputs, self.writes = ir, inputs, writes
        self.inventory, self.model, self.phase = inventory, model, phase
        self.neurons = neurons
        self.runtime_constants = runtime_constants or {}
        self.spike_pointers, self.spike_counts = set(), set()
        self.pointer_values = {}
        self.nodes, self.pointers, self.outputs, self.edges = {}, {}, {}, {}
        self.terminating_errors = []
        self.error_calls_remaining = len(_TERMINATING_ERROR.findall(ir))
        self.block = "0"
        self.paths = {"0": self.constant(True, "i1")}
        self.scalar, self.vector, self.temporaries, self.names = [], [], [], {}
        self.definitions = {match[1]: match[2] for line in ir.splitlines()
                            if (match := re.match(r"\s*(" + _REF + r") = (.*)", line))}
        self.uses = Counter()
        # Count the entire optimized function, not just its first unrolled row.
        for line in ir.splitlines():
            if line.strip().startswith("#dbg_") or "@llvm.dbg." in line:
                continue
            instruction = _pointwise(line.split(" = ", 1)[-1])
            self.uses.update(re.findall(_REF, instruction))
        self._parse()
        self._finish_terminating_errors()
        self.node_refs = {id(node): name for name, node in self.nodes.items()}

    def _finish_terminating_errors(self):
        """Transport selected Cython exception exits through native status.

        Keep the compiler's actual noreturn CFG during preparation. Replacing
        an exception exit by a bool store changes even preceding arithmetic
        instruction selection/NaN order. Execution still uses the existing
        native status, failed-row replay, clock and RNG machinery.
        """
        combined = self.constant(False, "i1")
        for role in ("scalar", "vector"):
            errors = [(condition, kind) for found, condition, kind in self.terminating_errors if found == role]
            if not errors:
                continue
            condition, kind = self.constant(False, "i1"), self.constant(0, "i32")
            for path, value in errors:
                condition = self.operation("or", (condition, path), "i1")
                kind = self.operation("select", (path, value, kind), "i32")
            self.outputs[f"_eon_{role}_error"] = condition
            self.outputs[f"_eon_{role}_error_kind"] = kind
            combined = self.operation("or", (combined, condition), "i1")
        missing = self.writes - self.outputs.keys()
        # A preparation-known unconditional exception eliminates all later
        # physical stores. Preserve their incoming values while native status
        # reports the error, including an empty spike selection's scalar exit.
        try:
            always_fails = bool(self.fold(combined))
        except (KeyError, ValueError, OverflowError, ZeroDivisionError):
            always_fails = False
        if missing and always_fails:
            for name in missing:
                variable = self.inventory.get(name)
                if variable is not None:
                    dtype = np.dtype(variable.dtype)
                    type = "double" if dtype.kind == "f" else "i1" if dtype.kind == "b" else "i" + str(dtype.itemsize * 8)
                    self.outputs[name] = _Node("input", (), type, variable.scalar, name)
                elif name == "_cond":
                    self.outputs[name] = self.constant(False, "i1")
                elif name in ("_eon_scalar_error", "_eon_vector_error"):
                    self.outputs[name] = self.constant(False, "i1")
                elif name in ("_eon_scalar_error_kind", "_eon_vector_error_kind"):
                    self.outputs[name] = self.constant(0, "i32")
        if self.writes - self.outputs.keys():
            raise UnsupportedModelError("CPU arithmetic outputs are missing: " + str(self.writes - self.outputs.keys()))

    @staticmethod
    def constant(value, type):
        return _Node("constant", (), type, True, value)

    def argument(self, value, type="double"):
        value = value.strip()
        if value.startswith("%"):
            if type == "ptr":
                return self.pointer_values[value]
            return self.nodes[value]
        if value in ("true", "false"):
            result = value == "true"
        elif type == "double":
            result = struct.unpack(">d", bytes.fromhex(value[2:]))[0] if value.startswith("0x") else float(value)
        else:
            result = int(value)
        return self.constant(result, type)

    def operation(self, op, args, type, value=None, flags=""):
        return _Node(op, tuple(args), type, all(a.shared for a in args), value, flags)

    def _edge(self, target, condition):
        self.edges[self.block, target] = condition
        old = self.paths.get(target)
        self.paths[target] = condition if old is None else self.operation("or", (old, condition), "i1")

    def _integer_control(self, node):
        return node.type != "double" and node.op != "input" and all(self._integer_control(a) for a in node.args)

    def _unreachable(self, node):
        if node is None:
            return False
        try:
            return not bool(self.fold(node))
        except (KeyError, ValueError, OverflowError, ZeroDivisionError):
            return False

    def _induction(self, name, type, incoming):
        if type not in ("i32", "i64"):
            return False
        initial, starts = False, {}
        pending = []
        for value, predecessor in incoming:
            value = value.strip()
            if value.startswith("%") and value not in self.nodes:
                pending.append((value, predecessor))
            else:
                try:
                    start = self.fold(self.argument(value, type))
                except KeyError:
                    return False
                if start < 0:
                    return False
                starts[predecessor] = start
                initial = True
        if not initial:
            return False
        for value, predecessor in pending:
            definition = self.definitions.get(value, "").split(", !", 1)[0]
            cast = re.match(r"(?:sext|zext) i32 (" + _REF + r") to i64$", definition)
            step_value = cast[1] if cast else value
            definition = self.definitions.get(step_value, "").split(", !", 1)[0]
            step = re.match(r"add (?:nuw |nsw )*i(?:32|64) (" + _REF + r"), [1-9]\d*$", definition)
            if step is None:
                return False
            if step[1] == name:
                continue
            # Clang can carry both an i32 row counter and its widened i64
            # address index. Prove they have the same starts/backedge, rather
            # than classifying an arbitrary recurrence as a neuron index.
            if not cast or self.neurons >= 2**31:
                return False
            other = self.definitions.get(step[1], "")
            if not other.startswith("phi i32 "):
                return False
            alternatives = dict((pred, val.strip()) for val, pred in
                re.findall(r"\[ ([^,]+), %([-\w.$]+) \]", other))
            if alternatives.get(predecessor) != step_value:
                return False
            for pred, start in starts.items():
                if pred not in alternatives or self.fold(self.argument(alternatives[pred], "i32")) != start:
                    return False
        return True

    def _parse(self):
        body = self.ir.split(f"define void @eon_cpu_{self.phase}()", 1)[1].split("\n}", 1)[0]
        for line in body.splitlines()[1:]:
            line = line.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("#dbg_") or "@llvm.dbg." in line:
                continue
            line = line.split(", !", 1)[0]
            label = re.match(r"([-\w.$]+):", line)
            if label:
                self.block = label[1]
                continue
            if self._unreachable(self.paths.get(self.block)):
                if _TERMINATING_ERROR.match(line):
                    self.error_calls_remaining -= 1
                # Do not require definitions from an impossible alias/trip
                # branch. Only preparation-known predicates are folded;
                # dynamic model branches retain their actual operation DAG.
                for target in re.findall(r"label %([-\w.$]+)", line):
                    self._edge(target, self.constant(False, "i1"))
                continue
            match = re.match(r"(" + _REF + r") = insertelement <\d+ x (double|i\d+)> poison, (?:double|i\d+) (\S+), i(?:32|64) 0$", line)
            if match:
                self.nodes[match[1]] = self.operation("partial_splat", (self.argument(match[3], match[2]),), match[2])
                continue
            match = re.match(r"(" + _REF + r") = shufflevector <\d+ x (double|i\d+)> (" + _REF + r"), <\d+ x (?:double|i\d+)> poison, <\d+ x i32> zeroinitializer$", line)
            if match:
                source = self.nodes[match[3]]
                if source.op != "partial_splat":
                    raise UnsupportedModelError("CPU vector broadcast does not originate in a scalar")
                self.nodes[match[1]] = self.operation("broadcast", source.args, match[2])
                continue
            line = _pointwise(line)
            match = _TERMINATING_ERROR.match(line)
            if match:
                role, value = match.groups()
                self.terminating_errors.append((role, self.paths.get(self.block, self.constant(True, "i1")),
                                                self.argument(value, "i32")))
                self.error_calls_remaining -= 1
                continue
            match = re.match(r"(" + _REF + r") = freeze (double|i\d+) (\S+)$", line)
            if match:
                self.nodes[match[1]] = self.operation("broadcast", (self.argument(match[3], match[2]),), match[2])
                continue
            match = re.match(r"(?:tail )?call void @llvm\.memset[^ (]*\(ptr[^%]*(" + _REF + r"), i8 ([^,]+),", line)
            if match and self.pointers.get(match[1], "").startswith("_eon_") and self.pointers[match[1]].endswith("_error"):
                self.outputs[self.pointers[match[1]]] = self.argument(match[2], "i8")
                if self.writes <= self.outputs.keys() and not self.error_calls_remaining:
                    return
                continue
            match = re.match(r"(" + _REF + r") = load (double|i\d+), ptr @([-\w.$]+),", line)
            if match:
                name, type, global_name = match.groups()
                self.nodes[name] = self.constant(self.runtime_constants[self.inputs[global_name]], type)
                continue
            match = re.match(r"br i1 (" + _REF + r"), label %([-\w.$]+), label %([-\w.$]+)", line)
            if match:
                condition = self.argument(match[1], "i1")
                path = self.paths.get(self.block, self.constant(True, "i1"))
                self._edge(match[2], self.operation("and", (path, condition), "i1"))
                self._edge(match[3], self.operation("and", (path, self.operation("not", (condition,), "i1")), "i1"))
                continue
            match = re.match(r"br label %([-\w.$]+)", line)
            if match:
                self._edge(match[1], self.paths.get(self.block, self.constant(True, "i1")))
                continue
            match = re.match(r"(" + _REF + r") = load ptr, ptr @([-\w.$]+),", line)
            if match:
                self.pointers[match[1]] = self.inputs[match[2]]
                # These symbolic addresses resolve compiler-generated overlap
                # guards only. Brian allocates distinct state arrays; native
                # execution also gives every variable a distinct logical store.
                base = (1 + list(self.inputs).index(match[2])) * (self.neurons*16 + 4096)
                self.pointer_values[match[1]] = self.constant(base, "i64")
                if match[2] == "_eon_spikes":
                    self.spike_pointers.add(match[1])
                continue
            match = re.match(r"(" + _REF + r") = getelementptr .*ptr (" + _REF + r"),", line)
            if match:
                self.pointers[match[1]] = self.pointers[match[2]]
                offset = re.search(r"getelementptr .*?(double|i\d+), ptr " + _REF + r", i(?:32|64) ([^,]+)", line)
                if offset:
                    size = {"double": 8, "i64": 8, "i32": 4, "i8": 1, "i1": 1}[offset[1]]
                    index = self.argument(offset[2], "i64")
                    amount = self.operation("mul", (index, self.constant(size, "i64")), "i64")
                    self.pointer_values[match[1]] = self.operation("add", (self.pointer_values[match[2]], amount), "i64")
                if match[2] in self.spike_pointers:
                    self.spike_pointers.add(match[1])
                    if offset:
                        try:
                            is_count = self.fold(index) == self.neurons
                        except KeyError:
                            is_count = False
                        if is_count:
                            self.spike_counts.add(match[1])
                continue
            match = re.match(r"(" + _REF + r") = load (double|i\d+), ptr (" + _REF + r"),", line)
            if match:
                name, type, pointer = match.groups()
                var = self.pointers[pointer]
                variable = self.inventory.get(var)
                # The native caller enters a reset only for a firing neuron.
                # The CPU spike-list loop therefore has an active iteration;
                # its trip-count predicate is true, while its row index is i.
                self.nodes[name] = (self.constant(1, type) if pointer in self.spike_counts else
                    self.constant(self.model.dt_s, type) if var == "dt" else
                    _Node("input", (), type, var in ("t", "t_in_timesteps") or bool(variable and variable.scalar), var))
                continue
            match = re.match(r"store (double|i\d+) ([^,]+), ptr (" + _REF + r"),", line)
            if match:
                type, value, pointer = match.groups()
                var = self.pointers[pointer]
                reachable = not self._unreachable(self.paths.get(self.block))
                if reachable and var in self.writes and var not in self.outputs:
                    self.outputs[var] = self.argument(value, type)
                if self.writes <= self.outputs.keys() and not self.error_calls_remaining:
                    return
                continue
            match = re.match(r"(" + _REF + r") = (fadd|fsub|fmul|fdiv|frem|add|sub|mul|sdiv|and|or|xor|shl|lshr|ashr|udiv) (.*?)(double|i\d+) ([^,]+), (\S+)", line)
            if match:
                name, op, flags, type, a, b = match.groups()
                self.nodes[name] = self.operation(op, (self.argument(a, type), self.argument(b, type)), type, flags=flags)
                continue
            match = re.match(r"(" + _REF + r") = fneg (.*?)double (\S+)", line)
            if match:
                self.nodes[match[1]] = self.operation("neg", (self.argument(match[3]),), "double", flags=match[2])
                continue
            match = re.match(r"(" + _REF + r") = (fptosi|sitofp|uitofp|zext|sext|trunc) (?:(?:nneg|nuw|nsw) )*(double|i\d+) (\S+) to (double|i\d+)", line)
            if match:
                name, op, source, value, target = match.groups()
                self.nodes[name] = self.operation(op, (self.argument(value, source),), target)
                continue
            match = re.match(r"(" + _REF + r") = (?:tail )?call .*?(i\d+) @llvm\.(scmp|ucmp|smin|smax|umin|umax)\.i\d+(?:\.i\d+)?\((i\d+) ([^,]+), i\d+ ([^)]+)\)", line)
            if match:
                name, target, operation, source, a, b = match.groups()
                self.nodes[name] = self.operation(operation, (self.argument(a, source), self.argument(b, source)), target)
                continue
            match = re.match(r"(" + _REF + r") = (?:tail )?call (.*?)double @([-\w.]+)\(([^)]*)\)", line)
            if match:
                name, flags, function, arguments = match.groups()
                typed=[re.match(r'(double|i\d+) (?:(?:noundef|signext|zeroext) )*(.*)',arg.strip())
                       for arg in arguments.split(',')]
                if any(item is None for item in typed):raise UnsupportedModelError('Unknown floating intrinsic argument '+arguments)
                args=[self.argument(item[2],item[1]) for item in typed]
                function = function.removeprefix("llvm.").removesuffix(".f64")
                self.nodes[name] = self.operation(function, args, "double", flags=flags)
                continue
            match = re.match(r"(" + _REF + r") = (fcmp|icmp) (.*?)(\w+) (double|i\d+|ptr) ([^,]+), (\S+)", line)
            if match:
                name, op, flags, predicate, type, a, b = match.groups()
                self.nodes[name] = self.operation(op, (self.argument(a, type), self.argument(b, type)), "i1", predicate, flags)
                continue
            match = re.match(r"(" + _REF + r") = select (.*?)i1 ([^,]+), (double|i\d+) ([^,]+), (double|i\d+) (\S+)", line)
            if match:
                name, flags, condition, type, a, _, b = match.groups()
                self.nodes[name] = self.operation("select", (self.argument(condition, "i1"), self.argument(a, type), self.argument(b, type)), type, flags=flags)
                continue
            match = re.match(r"(" + _REF + r") = phi (double|i\d+) (.*)", line)
            if match:
                name, type, alternatives = match.groups()
                incoming = re.findall(r"\[ ([^,]+), %([-\w.$]+) \]", alternatives)
                incoming = [(value, pred) for value, pred in incoming
                            if not self._unreachable(self.edges.get((pred, self.block), self.paths.get(pred)))]
                if any(value.strip().startswith("%") and value.strip() not in self.nodes for value, _ in incoming):
                    # Loop index: no cross-neuron state recurrence is legal in
                    # Brian's vector statements. Array GEPs consume this index.
                    if not self._induction(name, type, incoming):
                        raise UnsupportedModelError("Unresolved CPU arithmetic recurrence: " + line)
                    self.nodes[name] = _Node("input", (), type, False, "i")
                    continue
                value, predecessor = incoming[0]
                result = self.argument(value, type)
                for value, predecessor in incoming[1:]:
                    condition = self.edges.get((predecessor, self.block), self.paths.get(predecessor))
                    if condition is None:
                        raise UnsupportedModelError("Unresolved CPU arithmetic control-flow edge")
                    result = self.operation("select", (condition, self.argument(value, type), result), type)
                self.nodes[name] = result
                continue
            if line.startswith("ret ") or line == "unreachable":
                continue
            raise UnsupportedModelError(f"CPU arithmetic lowering does not implement {line}")

    def _fused(self, node):
        if node.op not in ("fadd", "fsub") or "contract" not in node.flags:
            return None
        for index, product in enumerate(node.args):
            if product.op != "fmul" or "contract" not in product.flags:
                continue
            # Constant CFG predicates can be folded while SSA parsing is
            # still in progress. Use the current aliases until it finishes.
            reference = (self.node_refs.get(id(product)) if hasattr(self, "node_refs") else
                         next((name for name, value in reversed(self.nodes.items()) if value is product), None))
            if self.uses[reference] == 1:
                a, b = product.args
                c = node.args[1-index]
                if node.op == "fsub":
                    if index == 0:
                        c = self.operation("neg", (c,), "double")
                    else:
                        a = self.operation("neg", (a,), "double")
                return a, b, c
        return None

    def fold(self, node, inputs=None, *, bindings=None):
        if bindings is not None and node in bindings:
            return bindings[node]
        if node.op == "constant":
            return node.value
        if node.op == "input":
            if inputs is not None and node.value in inputs:
                return inputs[node.value]
            raise KeyError(node.value)
        if node.op == "broadcast":
            return self.fold(node.args[0], inputs, bindings=bindings)
        if node.op == "select":
            return self.fold(node.args[1] if self.fold(node.args[0], inputs, bindings=bindings)
                             else node.args[2], inputs, bindings=bindings)
        # A constant trip/alias predicate can exclude a branch even when an
        # earlier scalar exception predicate depends on runtime model state.
        if node.type == "i1" and node.op in ("and", "or"):
            known=[];unknown=None
            for argument in node.args:
                try:value=bool(self.fold(argument,inputs,bindings=bindings))
                except KeyError as error:unknown=error;continue
                if value == (node.op == "or"):return value
                known.append(value)
            if unknown is not None:raise unknown
            return all(known) if node.op == "and" else any(known)
        fused = self._fused(node)
        args = [self.fold(a, inputs, bindings=bindings) for a in (fused or node.args)]
        if fused or node.op in ("fmuladd", "fma"):
            function = ctypes.CDLL(None).fma
            function.argtypes, function.restype = [ctypes.c_double]*3, ctypes.c_double
            return function(*args)
        def to_integer():
            bits = int(node.type[1:])
            if not math.isfinite(args[0]) or not -(2**(bits-1)) <= args[0] < 2**(bits-1):
                raise OverflowError("CPU float-to-integer expression is outside its defined range")
            return int(args[0])
        def divide():
            with np.errstate(all="ignore"):
                return np.float64(args[0]) / np.float64(args[1])
        operations = {"fadd": lambda: args[0]+args[1], "fsub": lambda: args[0]-args[1],
            "fmul": lambda: args[0]*args[1], "fdiv": divide, "neg": lambda: -args[0],
            "add": lambda: args[0]+args[1], "sub": lambda: args[0]-args[1], "mul": lambda: args[0]*args[1],
            "and": lambda: args[0]&args[1], "or": lambda: args[0]|args[1], "xor": lambda: args[0]^args[1],
            "shl": lambda: args[0]<<args[1], "lshr": lambda: args[0]>>args[1],
            "ashr": lambda: args[0]>>args[1], "udiv": lambda: args[0]//args[1],
            "not": lambda: not args[0], "fptosi": to_integer, "sitofp": lambda: float(args[0]),
            "uitofp": lambda: float(int(args[0]) & ((1 << int(node.args[0].type[1:]))-1)),
            "zext": lambda: int(args[0]), "sext": lambda: -int(args[0]) if node.args[0].type == "i1" else int(args[0]),
            "trunc": lambda: int(args[0])}
        if node.op in operations:
            return operations[node.op]()
        if node.op == "icmp":
            predicate = node.value.lstrip("su")
            if node.value.startswith("u"):
                bits = int(node.args[0].type.removeprefix("i"))
                args = [value & ((1 << bits)-1) for value in args]
            return {"eq": lambda: args[0] == args[1], "ne": lambda: args[0] != args[1],
                    "ge": lambda: args[0] >= args[1], "gt": lambda: args[0] > args[1],
                    "le": lambda: args[0] <= args[1], "lt": lambda: args[0] < args[1]}[predicate]()
        if node.op == "fcmp":
            unordered = any(math.isnan(value) for value in args)
            predicate = node.value
            if predicate in ("ord", "uno"):
                return unordered if predicate == "uno" else not unordered
            if predicate in ("true", "false"):
                return predicate == "true"
            if unordered:
                return predicate.startswith("u")
            return {"eq": lambda: args[0] == args[1], "ne": lambda: args[0] != args[1],
                    "ge": lambda: args[0] >= args[1], "gt": lambda: args[0] > args[1],
                    "le": lambda: args[0] <= args[1], "lt": lambda: args[0] < args[1]}[predicate[1:]]()
        if node.op in ("scmp", "ucmp", "smin", "smax", "umin", "umax"):
            if node.op.startswith("u"):
                bits = int(node.args[0].type[1:])
                args = [value & ((1 << bits)-1) for value in args]
            if node.op.endswith("cmp"):
                return int(args[0] > args[1])-int(args[0] < args[1])
            return (min if node.op.endswith("min") else max)(args)
        if self.operator_registry is not None:
            folded=self.operator_registry.constant(node.op,node.type,args)
            if folded is not None:return folded
        function = getattr(math, {"fabs": "fabs"}.get(node.op, node.op), None)
        if function is not None:
            return function(*args)
        raise KeyError(node.op)

    def expression(self, node, emit):
        fused = self._fused(node)
        if self.operator_registry is not None:
            boundary=fused or (node.args if node.op in ('fmuladd','fma') else None)
            late=self.operator_registry.late_expression(node,boundary,emit)
            if late is not None:return late
        if fused:
            return "d_fma(" + ", ".join(emit(a) for a in fused) + ")"
        if node.op == "fmul" and "nsz" in node.flags:
            for index, argument in enumerate(node.args):
                if argument.op == "neg":
                    # AArch64 selects FNMUL for a negated runtime operand.
                    # Negate the rounded result, including its NaN sign;
                    # folding the operand first loses this distinction.
                    return "d_neg(d_mul(" + emit(argument.args[0]) + ", " + emit(node.args[1-index]) + "))"
        if node.op == "exp":
            # Unoptimized linear refractory updates retain exp(c*int(mask)).
            # It has two CPU-evaluated constants, selected by the actual mask.
            # Preserve this finite boolean domain without a dynamic exp kernel.
            booleans = set()
            def finite(item):
                try:
                    self.fold(item)
                    return True
                except KeyError:
                    pass
                if item.type == "i1":
                    booleans.add(item)
                    return True
                return item.op != "input" and all(finite(a) for a in item.args)
            if finite(node) and len(booleans) == 1:
                boolean = next(iter(booleans))
                yes = _literal(self.fold(node, bindings={boolean: True}), "float")
                no = _literal(self.fold(node, bindings={boolean: False}), "float")
                return f"({emit(boolean)} ? {yes} : {no})"
        args = [emit(a) for a in node.args]
        if self.operator_registry is not None:
            code=self.operator_registry.intrinsic(node.op,'float64' if node.type=='double' else node.type,args)
            if code is not None:return code
        # LLVM already spells reciprocal hoisting as fdiv(1, d) + fmul.
        # The arcp flag permits that optimization; it does not require it.
        if node.op in ("fadd", "fsub", "fmul", "fdiv"):
            return {"fadd": "d_add", "fsub": "d_sub", "fmul": "d_mul", "fdiv": "d_div"}[node.op] + "(" + ", ".join(args) + ")"
        if node.op in ("neg", "fabs", "sqrt", "fma", "fmuladd"):
            return {"neg": "d_neg", "fabs": "d_abs", "sqrt": "d_sqrt", "fma": "d_fma", "fmuladd": "d_fma"}[node.op] + "(" + ", ".join(args) + ")"
        if node.op == "fptosi":
            return "d_to_int(" + args[0] + ")"
        if node.op == "sitofp":
            old=node.args[0].type
            value=(f"({'char' if old=='i8' else 'short' if old=='i16' else 'int'})({args[0]})"
                   if old in ('i8','i16','i32') else args[0])
            return "d_from_int((long)(" + value + "))"
        if node.op == "uitofp":
            bits = int(node.args[0].type[1:])
            value = "ulong(" + args[0] + ")"
            if bits < 64:
                value = f"({value} & {(1 << bits)-1}ul)"
            return "d_from_uint(" + value + ")"
        if node.op in ("zext", "sext", "trunc"):
            if node.op == 'trunc' and node.type == 'i1':
                return f'bool(ulong({args[0]}) & 1ul)'
            old=node.args[0].type
            value=args[0]
            if node.op=='sext' and old in ('i8','i16','i32'):
                value=f"({'char' if old=='i8' else 'short' if old=='i16' else 'int'})({value})"
            if node.op=='zext' and old!='i64':
                value=f"(ulong({value}) & {(1<<int(old[1:]))-1}ul)"
            cast = _TYPE[node.type][1] + "(" + value + ")"
            return "(-" + cast + ")" if node.op == "sext" and node.args[0].type == "i1" else cast
        if node.op == "select":
            return f"({args[0]} ? {args[1]} : {args[2]})"
        if node.op == "not":
            return "(!" + args[0] + ")"
        if node.op in ("add", "sub", "mul", "sdiv", "and", "or", "xor"):
            symbol = {"add": "+", "sub": "-", "mul": "*", "sdiv": "/", "and": "&", "or": "|", "xor": "^"}[node.op]
            return f"({args[0]} {symbol} {args[1]})"
        if node.op in ("scmp", "ucmp", "smin", "smax", "umin", "umax"):
            if node.op.startswith("u"):
                cast = "ulong" if node.args[0].type == "i64" else "uint"
                args = [f"{cast}({arg})" for arg in args]
            if node.op.endswith("cmp"):
                return f"(int({args[0]} > {args[1]}) - int({args[0]} < {args[1]}))"
            return ("min" if node.op.endswith("min") else "max") + "(" + ", ".join(args) + ")"
        if node.op == "icmp":
            predicate = node.value
            symbol = {"eq": "==", "ne": "!=", "sge": ">=", "sgt": ">", "sle": "<=", "slt": "<"}.get(predicate)
            if symbol:
                return f"({args[0]} {symbol} {args[1]})"
            symbol = {"uge": ">=", "ugt": ">", "ule": "<=", "ult": "<"}.get(predicate)
            if symbol:
                cast = "ulong" if node.args[0].type == "i64" else "uint"
                return f"({cast}({args[0]}) {symbol} {cast}({args[1]}))"
        if node.op == "fcmp":
            function = {"ogt": "d_gt", "oge": "d_ge", "olt": "d_lt", "ole": "d_le", "oeq": "d_eq", "une": "d_eq"}.get(node.value)
            if function:
                result = function + "(" + ", ".join(args) + ")"
                return "(!" + result + ")" if node.value == "une" else result
            inverse = {"ugt": "d_le", "uge": "d_lt", "ult": "d_ge", "ule": "d_gt"}.get(node.value)
            if inverse:
                return "(!" + inverse + "(" + ", ".join(args) + "))"
            nan = f"(d_is_nan({args[0]}) || d_is_nan({args[1]}))"
            if node.value == "ueq":
                return f"(d_eq({args[0]}, {args[1]}) || {nan})"
            if node.value == "one":
                return f"(!d_eq({args[0]}, {args[1]}) && !{nan})"
            if node.value in ("uno", "ord"):
                return nan if node.value == "uno" else "(!" + nan + ")"
        raise UnsupportedModelError(f"CPU arithmetic operation {node.op}/{node.value} is not implemented")

    def emit(self, node):
        try:
            return _literal(self.fold(node), _kind(node.type))
        except KeyError:
            pass
        if node.op == "broadcast":
            # Keep the broadcast node in the DAG (FMA cannot cross a scalar
            # CPU register duplication), but share its stored scalar value.
            return self.emit(node.args[0])
        if node in self.names:
            return self.names[node]
        if node is getattr(self, "guard_node", None):
            name = f"_eon_cpu_{self.phase}_{len(self.names)}"
            self.names[node] = name
            rfc = self.inventory["rfc"].code_name
            predicate = self.expression(node, self.inline)
            fast = (f"d_eq({rfc}, 0ul) || (lastspike == {_literal(-10000., 'float')}"
                    f" && d_eq({rfc}, {_literal(self.model.t_rfc, 'float')}))")
            self.vector.append(f"bool {name};\nif(FAST_CPU_REFRACTORY && ({fast})) {name}=true;"
                               f"\nelse {name}={predicate};")
            return name
        if node.op == "input":
            variable = self.inventory.get(node.value)
            code = variable.code_name if variable is not None else node.value
            if not (node.shared and node.value in self.writes):
                return code
        else:
            code = self.expression(node, self.emit)
        name = f"_eon_cpu_{self.phase}_{len(self.names)}"
        dtype, metal = _TYPE[node.type]
        self.names[node] = name
        target = self.scalar if node.shared else self.vector
        target.append(f"{metal} {name} = {code};")
        if node.shared:
            self.temporaries.append(CompiledVariable(name, dtype, metal, True, True, True, None, False))
        return name

    def inline(self, node):
        if node.op == "broadcast":
            return self.inline(node.args[0])
        if node.shared or node.op == "input":
            return self.emit(node)
        return self.expression(node, self.inline)

    def refractory_guard(self):
        """Prove the two cheap cases from the actual rounded CPU predicate."""
        if np.ndim(self.model.t_rfc) != 0 or isinstance(self.model.t_rfc, str):
            return None
        node = self.outputs.get("not_refractory")
        if node is None:
            return None
        while node.op in ("zext", "sext", "trunc", "broadcast"):
            node = node.args[0]
        if node.op != "icmp" or node.value != "sge":
            return None

        def tree(item):
            if item.op == "broadcast":
                return tree(item.args[0])
            try:
                value = self.fold(item)
            except KeyError:
                pass
            else:
                return ("constant", value)
            if item.op == "input":
                return ("input", item.value)
            if self._fused(item) is not None:
                return ("fma",)
            return (item.op, *(tree(a) for a in item.args))

        def quantized(item):
            if item.op == "broadcast":
                return quantized(item.args[0])
            if item.op != "fptosi" or item.type != "i64":
                return None
            value = tree(item.args[0])
            if value[0] == "fdiv" and value[2][0] == "constant":
                return value[1], ("fdiv", value[2][1])
            if value[0] == "fmul":
                for index in (1, 2):
                    if value[index][0] == "constant":
                        return value[3-index], ("fmul", value[index][1])
            return None

        left, right = (quantized(arg) for arg in node.args)
        if left is None or right is None or left[1] != right[1]:
            return None
        if not math.isfinite(left[1][1]) or left[1][1] <= 0:
            return None
        offset = ("constant", self.model.dt_s*.001)
        t, last, rfc = (("input", name) for name in ("t", "lastspike", "rfc"))
        add = lambda a, b: ("fadd", a, b)
        starts = [("fsub", add(t, offset), last), ("fsub", add(offset, t), last),
                  add(("fsub", t, last), offset), add(offset, ("fsub", t, last))]
        if left[0] not in starts or right[0] not in (add(rfc, offset), add(offset, rfc)):
            return None
        try:
            duration_zero = self.fold(node.args[1], {"rfc": 0.})
            initial = self.fold(node, {"t": 0., "lastspike": -10000., "rfc": self.model.t_rfc})
        except (KeyError, OverflowError, ValueError):
            return None
        # Both recognized numerator orders are monotone in t. rfc0 gives
        # duration0 and elapsed>=0 for threshold-written lastspike<=t. The
        # sentinel case's t=0 minimum is checked in the exact CPU order above.
        return node if duration_zero == 0 and initial else None

    def render(self, *, guard=False):
        self.scalar, self.vector, self.temporaries, self.names = [], [], [], {}
        self.guard_node = self.refractory_guard() if guard else None
        assignments = []
        for name, node in self.outputs.items():
            if name in ("_eon_scalar_error", "_eon_vector_error","_eon_scalar_error_kind","_eon_vector_error_kind"):
                continue
            variable = self.inventory.get(name)
            code = variable.code_name if variable is not None else name
            assignments.append((variable is not None and variable.scalar, code + " = " + self.emit(node) + ";"))
        scalar_errors, vector_error = [], False
        for role in ("scalar", "vector"):
            condition = self.outputs.get(f"_eon_{role}_error")
            if condition is None:
                continue
            if self._unreachable(condition):
                continue
            code = self.emit(condition)
            if self.operator_registry is not None:
                kind=self.emit(self.outputs[f'_eon_{role}_error_kind'])
                (self.scalar if role=='scalar' else self.vector).append(self.operator_registry.status(code,kind))
                continue
            if role == "scalar":
                name = f"_eon_cpu_{self.phase}_error_{len(scalar_errors)}"
                self.scalar.append(f"bool {name} = {code};")
                self.temporaries.append(CompiledVariable(name, "bool", "bool", True, True, True, None, False))
                scalar_errors.append(name)
            else:
                self.vector.append(f"if ({code}) _eon_cython_error = true;")
                vector_error = True
        for shared, line in assignments:
            (self.scalar if shared else self.vector).append(line)
        return ScalarStage("\n".join(self.scalar), tuple(self.temporaries), "\n".join(self.vector),
                           "\n".join(self.vector), tuple(scalar_errors), vector_error)


def lower_stage(model, inventory, resolved, scalar, vector, group, runner, phase, neurons, identity, target,operator_registry=None):
    source, inputs, writes, runtime_constants = _source(resolved, scalar, vector, group, runner, phase, neurons, target,operator_registry)
    ir = _optimize(source, identity)
    graph = _Graph(ir, inputs, writes, inventory, model, phase, neurons, runtime_constants,operator_registry)
    return graph, hashlib.sha256(source.encode()).hexdigest()
