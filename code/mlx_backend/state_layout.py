"""A bounded Metal binding layout for typed persistent neuron state.

Each dtype gets one planar buffer: variable slots contain contiguous complete
trial/neuron rows. The number of bindings depends on supported dtypes, not on
the number of model variables. Binary64 state retains its raw uint64 storage.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class StateGroup:
    dtype: str
    names: tuple[str, ...]

    @property
    def buffer_name(self):
        return "packed_" + self.dtype


@dataclass(frozen=True)
class StateLayout:
    groups: tuple[StateGroup, ...]
    names: tuple[str, ...]
    slots: Mapping[str, tuple[int, int]]
    dtypes: Mapping[str,str]

    @classmethod
    def from_program(cls, program, *, scalar=None):
        variables = tuple(v for v in program.variables.values() if v.persistent
                          and (scalar is None or v.scalar == scalar))
        grouped = {}
        for variable in variables:
            grouped.setdefault(variable.storage_dtype, []).append(variable.name)
        # Five or more distinct declared dtypes can exceed Metal's31slots
        # once event, error, namespace and Poisson bindings are included.
        # Equal-width storage shares raw unsigned planes; named views retain
        # every original field dtype, with no numeric conversion or widening.
        if len(grouped)>4:
            grouped={}
            for variable in variables:
                dtype='uint'+str(np.dtype(variable.dtype).itemsize*8)
                grouped.setdefault(dtype,[]).append(variable.name)
        groups = tuple(StateGroup(dtype, tuple(names)) for dtype, names in grouped.items())
        slots = {name: (group, slot) for group, entry in enumerate(groups)
                 for slot, name in enumerate(entry.names)}
        return cls(groups, tuple(v.name for v in variables), MappingProxyType(slots),
                   MappingProxyType({v.name:v.storage_dtype for v in variables}))

    def decode(self,program,name,expression):
        variable=program.variables[name]
        group,_=self.slots[name]
        if self.groups[group].dtype==variable.dtype:return expression
        if variable.metal_type=='bool':return f'bool({expression})'
        return f'as_type<{variable.metal_type}>({expression})'

    def encode(self,program,name,expression):
        variable=program.variables[name]
        group,_=self.slots[name]
        dtype=self.groups[group].dtype
        if dtype==variable.dtype:return expression
        metal={'uint8':'uchar','uint16':'ushort','uint32':'uint','uint64':'ulong'}[dtype]
        return f'{metal}({expression})' if variable.metal_type=='bool' else f'as_type<{metal}>({expression})'

    def shapes(self, size):
        # MLX's one-dimensional array shape is a signed int. Check before any
        # allocation or generated offset can overflow it.
        shapes = tuple((len(group.names) * size,) for group in self.groups)
        if any(shape[0] > 2**31 - 1 for shape in shapes):
            raise OverflowError("Packed state buffer exceeds MLX array indexing")
        return shapes

    def source(self, program, output_index="gid", size="(NEURONS * B)", aliases=None):
        loads, stores = [], []
        for name in self.names:
            variable = program.variables[name]
            local = (aliases or {}).get(name, variable.code_name)
            group, slot = self.slots[name]
            buffer = self.groups[group].buffer_name
            # Brian reserves names starting with '_' for generated variables.
            # Keep buffer arguments in that namespace so valid state names
            # such as p_packed_float64 cannot shadow an argument.
            raw=f"_eon_p_{buffer}[{slot}ul * {size} + gid]"
            loads.append(f"{variable.metal_type} {local} = {self.decode(program,name,raw)};")
            stores.append(f"_eon_q_{buffer}[{slot}ul * {size} + {output_index}] = {self.encode(program,name,local)};")
        return "\n".join(loads), "\n".join(stores)
