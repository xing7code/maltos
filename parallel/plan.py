from __future__ import annotations

from dataclasses import dataclass, field

from .context import ContextParallelAttentionCoreType
from .schedule import PipelineScheduleConfig


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
