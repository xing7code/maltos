from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TpSpShardAxis(str, Enum):
    PARAM_OUT = "param_out"
    PARAM_IN = "param_in"
    SEQUENCE = "sequence"


class TpSpComm(str, Enum):
    NONE = "none"
    ALL_GATHER = "all_gather"
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    SCATTER = "scatter"


@dataclass(frozen=True)
class TpSpShardRule:
    module_path: str
    shard_axis: TpSpShardAxis | str
    pre_comm: TpSpComm | str = TpSpComm.NONE
    post_comm: TpSpComm | str = TpSpComm.NONE
    comm_dim: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(self, "shard_axis", TpSpShardAxis(self.shard_axis))
        object.__setattr__(self, "pre_comm", TpSpComm(self.pre_comm))
        object.__setattr__(self, "post_comm", TpSpComm(self.post_comm))


@dataclass(frozen=True)
class TpSpParallelSpec:
    rules: list[TpSpShardRule]
    tie_rules: list[tuple[str, str]] = field(default_factory=list)
