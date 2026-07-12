from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .context_interfaces import ContextParallelAttentionCoreType


class PipelineScheduleType(str, Enum):
    ONE_FWD_ONE_BWD = "1f1b"


@dataclass(frozen=True)
class PipelineScheduleConfig:
    schedule: PipelineScheduleType = PipelineScheduleType.ONE_FWD_ONE_BWD
    microbatches: int = 1

    def __post_init__(self) -> None:
        if self.microbatches < 1:
            raise ValueError(f"microbatches must be >= 1, got {self.microbatches}")


@dataclass(frozen=True)
class ParallelPlan:
    cp_attn_core: ContextParallelAttentionCoreType = ContextParallelAttentionCoreType.ALL_GATHER_KV
    pp_schedule: PipelineScheduleConfig = field(default_factory=PipelineScheduleConfig)
    # EP dimension reuse: whether EP groups borrow TP/CP ranks before spilling into DP.
    # reuse_tp_for_ep requires SP to be enabled (TP ranks hold duplicate tokens otherwise).
    # reuse_cp_for_ep allows EP groups to span CP ranks when EP > TP.
    # If both enabled, EP use ranks in priority TP > CP > DP.
    reuse_tp_for_ep: bool = True
    reuse_cp_for_ep: bool = True
